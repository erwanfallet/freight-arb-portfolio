"""Data contract enforcement.

Rule (see PROJECT_NOTES.md / brainstorm Partie 6.7): no series enters a calculation
without a filled-in contract. The stale-BSI-shown-as-current failure that killed the
prior attempt was a contract failure, not a modeling failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("ticker", "description", "native_unit", "frequency", "source", "gap_policy")
VALID_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly", "yearly", "irregular"}
VALID_GAP_POLICIES = {"none"}  # forward-fill is deliberately not a valid policy — gaps stay gaps


class ContractViolation(Exception):
    """Raised when a series or its dictionary entry fails the data contract."""


@dataclass(frozen=True)
class SeriesContract:
    ticker: str
    description: str
    native_unit: str
    frequency: str
    source: str
    gap_policy: str = "none"
    first_valid_date: date | None = None
    last_valid_date: date | None = None

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ContractViolation("ticker must not be empty")
        if not self.native_unit:
            raise ContractViolation(f"{self.ticker}: native_unit must not be empty")
        if self.frequency not in VALID_FREQUENCIES:
            raise ContractViolation(
                f"{self.ticker}: frequency '{self.frequency}' not in {VALID_FREQUENCIES}"
            )
        if not self.source:
            raise ContractViolation(f"{self.ticker}: source must not be empty")
        if self.gap_policy not in VALID_GAP_POLICIES:
            raise ContractViolation(
                f"{self.ticker}: gap_policy '{self.gap_policy}' not allowed — "
                "a gap is information, it is never filled before the audit step"
            )


def load_dictionary(path: str | Path) -> dict[str, SeriesContract]:
    """Load the ticker dictionary (data_dictionary.csv) into ticker -> SeriesContract."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ContractViolation(f"dictionary {path} is missing columns: {sorted(missing)}")

    contracts: dict[str, SeriesContract] = {}
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        contracts[ticker] = SeriesContract(
            ticker=ticker,
            description=str(row["description"]),
            native_unit=str(row["native_unit"]),
            frequency=str(row["frequency"]).strip().lower(),
            source=str(row["source"]),
            gap_policy=str(row.get("gap_policy", "none")).strip().lower(),
            first_valid_date=_parse_date(row.get("first_valid_date")),
            last_valid_date=_parse_date(row.get("last_valid_date")),
        )
    return contracts


def _parse_date(value) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return pd.to_datetime(value).date()


def validate_series(df: pd.DataFrame, contract: SeriesContract, *, today: date | None = None) -> None:
    """Raise ContractViolation if a long-format series (date, ticker, valeur) violates its contract.

    Deliberately does NOT check for "completeness" in the sense of no gaps — gaps are
    allowed and expected. It checks for the failure modes named explicitly in the
    brainstorm: unknown ticker, wrong ticker in file, and stale data silently presented
    as current.
    """
    required_cols = {"date", "ticker", "valeur"}
    if not required_cols.issubset(df.columns):
        raise ContractViolation(
            f"{contract.ticker}: series must be long-format with columns {required_cols}, "
            f"got {set(df.columns)}"
        )

    series = df[df["ticker"] == contract.ticker]
    if series.empty:
        raise ContractViolation(f"{contract.ticker}: no rows for this ticker in the given frame")

    dates = pd.to_datetime(series["date"])
    last_obs = dates.max().date()

    if contract.last_valid_date is not None and last_obs != contract.last_valid_date:
        raise ContractViolation(
            f"{contract.ticker}: dictionary says last_valid_date={contract.last_valid_date} "
            f"but the data's actual last observation is {last_obs} — re-check the dictionary "
            "before trusting this series (this is the exact BSI-2017 failure mode)"
        )

    today = today or datetime.utcnow().date()
    staleness_days = (today - last_obs).days
    # Flag, do not silently pass: a series can legitimately be stale (e.g. a suspended
    # index) but the caller must have explicitly recorded that in last_valid_date.
    if contract.last_valid_date is None and staleness_days > 0:
        raise ContractViolation(
            f"{contract.ticker}: last_valid_date is not set in the dictionary but the "
            f"series' latest observation ({last_obs}) is {staleness_days}d old — "
            "fill in last_valid_date explicitly so staleness is a decision, not an accident"
        )
