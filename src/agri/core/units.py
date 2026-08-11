"""Unit conversions — the number one cause of a wrong page in agri.

The principle governing this module: **a single physical constant**, `LB_PER_TONNE`,
from which everything else is derived by calculation. Writing 36.7437 and 22.0462 side
by side as two independent numbers is a guarantee that one day one of them gets
corrected and not the other. Here, fixing the tonne fixes everything.

THE TRAP, AND IT'S THE SUBJECT OF SEVERAL PAGES
------------------------------------------------
The CBOT board crush mixes three units into a single formula:

    bean   -> USD/bushel   (soybean bushel = 60 lb)
    meal   -> USD/short ton (2000 lb, NOT a metric tonne)
    oil    -> cents/lb

This is the same family of trap as dmt/wmt (iron ore) or gallon/tonne (distillate): the
quoted unit is not the economic unit. See `docs/NOUVELLE_CHAINE.md`.

THREE VERIFIED POINTS, NOT TO BE RE-DERIVED EVERY TIME
--------------------------------------------------------
U-H1  The bushel is not a usable volume unit here: it's a **test weight** that depends on
      the seed (soybean and wheat 60 lb, corn and sorghum 56 lb, oats 32 lb, barley
      48 lb). A bushel->tonne factor that doesn't name the seed is a bug.
U-H2  NY No.11 sugar is in c/lb on a **96° pol basis**, white No.5 in USD/t. Comparing
      them requires both the conversion AND a polarisation adjustment — `pol_adjust` is
      a parameter, never a fixed constant (see T2-4).
U-H3  Chinese VAT on imported oilseeds is not the same as on processed products, and the
      rates move. `strip_vat` takes the rate AND forces the caller to know which product
      and which date it's talking about.
"""
from __future__ import annotations

from typing import TypeVar

import pandas as pd

Numeric = TypeVar("Numeric", float, pd.Series)

# ---------------------------------------------------------------------------
# The module's only physical constant. Everything else is derived from it.
# ---------------------------------------------------------------------------
LB_PER_TONNE = 2204.62262      # avoirdupois pounds in a metric tonne
LB_PER_SHORT_TON = 2000.0
KG_PER_LB = 0.45359237         # exact legal definition of the pound
LITRES_PER_US_GALLON = 3.785411784   # exact definition of the US gallon

# ---------------------------------------------------------------------------
# Test weight by seed, in pounds per bushel (U-H1).
# Source: USDA standards. These are regulatory definitions, not measurements.
# ---------------------------------------------------------------------------
BUSHEL_WEIGHT_LB: dict[str, float] = {
    "soybean": 60.0,
    "wheat": 60.0,
    "corn": 56.0,
    "sorghum": 56.0,
    "rye": 56.0,
    "barley": 48.0,
    "oats": 32.0,
}


class UnitError(ValueError):
    """Impossible or mis-specified conversion — always a caller error."""


# ---------------------------------------------------------------------------
# Grains: bushel <-> metric tonne
# ---------------------------------------------------------------------------
def bushels_per_tonne(commodity: str) -> float:
    """Number of bushels in a metric tonne, for a named seed.

    >>> round(bushels_per_tonne("soybean"), 4)
    36.7437
    >>> round(bushels_per_tonne("corn"), 4)
    39.3683
    """
    key = commodity.strip().lower()
    if key not in BUSHEL_WEIGHT_LB:
        raise UnitError(
            f"unknown test weight for '{commodity}' — known seeds: "
            f"{sorted(BUSHEL_WEIGHT_LB)}. Don't guess: the factor depends on the seed."
        )
    return LB_PER_TONNE / BUSHEL_WEIGHT_LB[key]


def usd_bu_to_usd_t(price: Numeric, commodity: str) -> Numeric:
    """USD/bushel -> USD/metric tonne. The seed is mandatory (U-H1)."""
    return price * bushels_per_tonne(commodity)


def usd_t_to_usd_bu(price: Numeric, commodity: str) -> Numeric:
    """USD/metric tonne -> USD/bushel."""
    return price / bushels_per_tonne(commodity)


# ---------------------------------------------------------------------------
# Cents and short-ton quotes
# ---------------------------------------------------------------------------
def cents_lb_to_usd_t(price: Numeric) -> Numeric:
    """c/lb -> USD/t. Sugar, cotton, arabica coffee, soybean oil.

    >>> round(cents_lb_to_usd_t(1.0), 4)
    22.0462
    """
    return price * (LB_PER_TONNE / 100.0)


def usd_t_to_cents_lb(price: Numeric) -> Numeric:
    """USD/t -> c/lb."""
    return price / (LB_PER_TONNE / 100.0)


def usd_short_ton_to_usd_t(price: Numeric) -> Numeric:
    """USD/short ton -> USD/metric tonne. **CBOT meal is quoted in short tons.**

    >>> round(usd_short_ton_to_usd_t(1.0), 5)
    1.10231
    """
    return price * (LB_PER_TONNE / LB_PER_SHORT_TON)


def usd_t_to_usd_short_ton(price: Numeric) -> Numeric:
    """USD/metric tonne -> USD/short ton."""
    return price / (LB_PER_TONNE / LB_PER_SHORT_TON)


# ---------------------------------------------------------------------------
# Mass and volume
# ---------------------------------------------------------------------------
def lb_to_kg(mass: Numeric) -> Numeric:
    return mass * KG_PER_LB


def kg_to_lb(mass: Numeric) -> Numeric:
    return mass / KG_PER_LB


def gallons_to_litres(volume: Numeric) -> Numeric:
    return volume * LITRES_PER_US_GALLON


def litres_to_gallons(volume: Numeric) -> Numeric:
    return volume / LITRES_PER_US_GALLON


# ---------------------------------------------------------------------------
# FX and tax
# ---------------------------------------------------------------------------
def local_to_usd_per_t(price_local: Numeric, fx_local_per_usd: Numeric) -> Numeric:
    """MYR/t, CNY/t, BRL/t -> USD/t.

    `fx_local_per_usd` is the **local-per-dollar** quote (USDBRL = 5.4 means 5.4 BRL
    per 1 USD), so it divides. That's the usual quoting convention for BRL, CNY and MYR;
    inverting it silently passes a price through a factor of 25 without raising an
    error, hence the plausibility check below.
    """
    if isinstance(fx_local_per_usd, float) and fx_local_per_usd <= 0:
        raise UnitError(f"exchange rate must be > 0, got {fx_local_per_usd}")
    return price_local / fx_local_per_usd


def strip_vat(price_incl: Numeric, vat_rate: float) -> Numeric:
    """VAT-inclusive price -> ex-VAT price. `vat_rate` as a fraction (0.09 = 9%).

    China: ~9% on agricultural goods, ~13% on industrial goods — but check the rate
    applicable **to the product and the date** (U-H3). This module doesn't choose for
    you.
    """
    if not 0.0 <= vat_rate < 1.0:
        raise UnitError(f"VAT rate must be in [0, 1), got {vat_rate}")
    return price_incl / (1.0 + vat_rate)


def add_vat(price_excl: Numeric, vat_rate: float) -> Numeric:
    """Ex-VAT price -> VAT-inclusive price."""
    if not 0.0 <= vat_rate < 1.0:
        raise UnitError(f"VAT rate must be in [0, 1), got {vat_rate}")
    return price_excl * (1.0 + vat_rate)


# ---------------------------------------------------------------------------
# Sugar: comparing No.11 / No.5 requires a polarisation adjustment (U-H2)
# ---------------------------------------------------------------------------
def raw_sugar_to_white_basis(
    ny11_cents_lb: Numeric, *, pol_adjust: float = 1.07
) -> Numeric:
    """NY No.11 (c/lb, 96° pol) -> USD/t on a refined-white basis.

    `pol_adjust` is the raw->white yield factor, around 1.06-1.08 depending on the
    specification used. **A parameter, not a constant**: fixing it would distort the
    entire T2-4 white premium, which is precisely the residual being measured.
    """
    if not 1.0 <= pol_adjust <= 1.20:
        raise UnitError(
            f"pol_adjust outside the plausible range [1.00, 1.20], got {pol_adjust} — "
            "beyond that, it's no longer a polarisation correction"
        )
    return cents_lb_to_usd_t(ny11_cents_lb) * pol_adjust


# ---------------------------------------------------------------------------
# CBOT board crush — the canonical example of three units in one formula
# ---------------------------------------------------------------------------
# Implied yields of the CBOT contract, per bushel of soybean crushed:
MEAL_LB_PER_BUSHEL = 44.0
OIL_LB_PER_BUSHEL = 11.0


def board_crush_usd_bu(
    bean_usd_bu: Numeric,
    meal_usd_short_ton: Numeric,
    oil_cents_lb: Numeric,
) -> Numeric:
    """CBOT board crush in USD/bushel, with all three units handled explicitly.

    Derived coefficients, never hard-coded:
        meal : 44 lb/bu / 2000 lb/short ton = 0.022 short ton per bushel
        oil  : 11 lb/bu, price in c/lb      -> /100 to convert to USD

    Dollars per bushel is kept as the output unit because that's what a board desk
    talks in. Use `usd_bu_to_usd_t(..., "soybean")` to compare against a plant crush
    quoted per tonne.
    """
    meal_short_tons_per_bu = MEAL_LB_PER_BUSHEL / LB_PER_SHORT_TON
    oil_usd_per_lb = oil_cents_lb / 100.0
    return meal_short_tons_per_bu * meal_usd_short_ton + OIL_LB_PER_BUSHEL * oil_usd_per_lb - bean_usd_bu
