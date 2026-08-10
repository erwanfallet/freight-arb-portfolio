"""Computed-results snapshot, so the deployed app can show the work without the data.

THE PROBLEM. Every real-data page in this portfolio reads a Bloomberg export that lives on
one machine and cannot be redistributed — the repository is public and the data is licensed.
Deployed as-is, twelve of thirteen pages would render an error and nothing else.

THE ANSWER, AND ITS LIMIT. What gets committed is not the market data: it is the **frames
this codebase computed from it** — crush margins, freight-implied TCEs, breakeven curves,
switching prices. They are the output of the analysis, at the default parameters, and they
are what a reader needs in order to see the argument. The raw series stay out.

The limit is real and worth stating rather than hiding: with no live data, the sliders have
nothing to recompute against, so a deployed page shows **one parameterisation** rather than
an interactive model. Pages detect this through `has_live_data()` and say so. Run the app
locally with the export in place and everything becomes live again, sliders included.

WHY NOT CACHE THE RAW SERIES INSTEAD. It would keep the sliders working, and it is exactly
what must not be done: a parquet of Bloomberg price series is the Bloomberg data, whatever
the file extension. The choice here costs interactivity and keeps the licence intact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agri.data.bloomberg_loader import DEFAULT_PATH

SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshot"


class SnapshotMissing(FileNotFoundError):
    """A page asked for a frame that is neither computable nor cached."""


def has_live_data() -> bool:
    """True when the Bloomberg export is present and everything can be recomputed."""
    return DEFAULT_PATH.exists()


def snapshot_path(name: str) -> Path:
    return SNAPSHOT_DIR / f"{name}.parquet"


def save_frame(name: str, frame: pd.DataFrame) -> Path:
    """Write one computed frame to the snapshot. Called only by `scripts/build_snapshot.py`."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(name)
    # attrs do not survive parquet; the frames that carry them re-derive them on load.
    frame.to_parquet(path)
    return path


def load_frame(name: str) -> pd.DataFrame:
    """Read one computed frame from the snapshot."""
    path = snapshot_path(name)
    if not path.exists():
        raise SnapshotMissing(
            f"no live Bloomberg export and no snapshot for {name!r}. Either place the "
            f"export at {DEFAULT_PATH} or run `python scripts/build_snapshot.py` on a "
            "machine that has it."
        )
    return pd.read_parquet(path)


def cached(name: str, *, to_frame=None, from_frame=None):
    """Decorator: compute live when the export is present, otherwise serve the snapshot.

    The wrapped function keeps its signature, so nothing downstream changes. When there is
    no live data the arguments are **ignored** — there is nothing to recompute against — and
    that is precisely why deployed pages freeze their sliders instead of pretending the
    parameters still do something.

    `to_frame` / `from_frame` handle the loaders that do not return a bare DataFrame. A
    Series or a small dataclass of Series is flattened to one frame on the way out and
    rebuilt on the way in, so parquet only ever has to store a table.
    """

    def decorate(function):
        def wrapper(*args, **kwargs):
            if has_live_data():
                return function(*args, **kwargs)
            frame = load_frame(name)
            return from_frame(frame) if from_frame is not None else frame

        wrapper.__name__ = function.__name__
        wrapper.__doc__ = function.__doc__
        wrapper.__wrapped__ = function
        wrapper.snapshot_name = name
        wrapper.to_frame = to_frame
        return wrapper

    return decorate


def to_snapshot_frame(value) -> pd.DataFrame:
    """Flatten whatever a loader returned into something parquet can hold."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame(value.name or "value")
    raise TypeError(
        f"{type(value).__name__} needs an explicit `to_frame` in its @cached decorator"
    )


def load_series(name: str, column: str | None = None) -> pd.Series:
    """One cached series, for the few places a page needs a raw leg rather than a frame."""
    frame = load_frame(name)
    return frame[column] if column else frame.iloc[:, 0]


def series_or_live(name: str, loader_key: str) -> pd.Series:
    """Live series when the export is present, cached one otherwise.

    Used by the handful of pages that reach past the derived loaders for a single leg.
    """
    if has_live_data():
        from agri.data.bloomberg_loader import load

        return load(loader_key)
    return load_series(name)


def available() -> list[str]:
    """Names of every frame currently in the snapshot."""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(path.stem for path in SNAPSHOT_DIR.glob("*.parquet"))


__all__ = [
    "SNAPSHOT_DIR",
    "SnapshotMissing",
    "available",
    "cached",
    "has_live_data",
    "load_frame",
    "load_series",
    "series_or_live",
    "save_frame",
    "snapshot_path",
]
