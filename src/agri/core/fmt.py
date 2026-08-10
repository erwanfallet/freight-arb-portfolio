"""Number formatting helpers shared by pages, in standard English convention.

Thin wrappers over Python's own `,.Nf` format spec — kept here so pages import one name
instead of repeating a format spec, and so `sign=True` reads clearly at the call site for
quantities whose direction is the message (a margin, a gap, a P&L).
"""
from __future__ import annotations


def fmt_num(value: float, decimals: int = 2, *, sign: bool = False) -> str:
    """Format a number: `fmt_num(1843.5, 1)` -> `1,843.5`.

    `sign=True` forces an explicit sign, for quantities whose direction is the message.
    """
    spec = f"{'+' if sign else ''},.{decimals}f"
    return format(float(value), spec)


def fmt_pct(value: float, decimals: int = 0) -> str:
    """Format a fraction as a percentage: `fmt_pct(0.653)` -> `65%`."""
    return f"{fmt_num(value * 100.0, decimals)}%"


__all__ = ["fmt_num", "fmt_pct"]
