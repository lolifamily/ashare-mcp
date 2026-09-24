"""Shared utilities — no baostock/akshare dependency."""

from __future__ import annotations

import datetime as dt
import math
import sys
import threading
from contextlib import contextmanager
from io import StringIO
from typing import TYPE_CHECKING, TextIO, cast

import numpy as np
import pandas as pd
from dateutil.tz import gettz

if TYPE_CHECKING:
    from collections.abc import Generator

_tz = gettz("Asia/Shanghai")
if _tz is None:
    _msg = "Asia/Shanghai timezone unavailable; is python-dateutil installed?"
    raise RuntimeError(_msg)
MARKET_TZ: dt.tzinfo = _tz

ZERO_THRESHOLD: float = 1e-9

type Record = dict[str, object]


class _Quiet:
    """Process-wide stdout/stderr mute, reference-counted across threads.

    baostock print()s login banners and validation errors, akshare draws tqdm
    progress bars; none of it belongs on the stdio JSON-RPC wire or in the log.

    Not contextlib.redirect_stdout/redirect_stderr: each of those remembers the
    stream it saw on entry and restores it on exit. mcp 2.x runs sync tools on
    worker threads, so overlapping redirects that exit out of order (parallel
    baostock and akshare calls) restore a stale value and leave both streams
    stuck on a dead StringIO -- every later log line vanishes. Here the real
    streams are saved once: the first block in installs the sink, the last one
    out restores them. The lock guards only that bookkeeping, never the block,
    so the wrapped network calls still run concurrently.

    Cost: while any block is active the whole process is muted, so other
    threads' output (the SDK's log lines included) is dropped too. Muting per
    thread would take a thread-dispatching proxy installed on sys.stdout and
    sys.stderr for good, visible to every library that reads them.
    """

    def __init__(self) -> None:
        """Start with no block active."""
        self._lock = threading.Lock()
        self._depth = 0
        self._saved: tuple[TextIO, TextIO] = (sys.stdout, sys.stderr)

    @contextmanager
    def __call__(self) -> Generator[None]:
        """Mute sys.stdout/sys.stderr until the last overlapping block exits."""
        with self._lock:
            if self._depth == 0:
                self._saved = (sys.stdout, sys.stderr)
                sys.stdout = sys.stderr = StringIO()
            self._depth += 1
        try:
            yield  # the wrapped call runs here, outside the lock
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    sys.stdout, sys.stderr = self._saved


quiet = _Quiet()


def lookback_range(days: int, *, end: str | None = None) -> tuple[str, str]:
    """Return (start, end) 'YYYY-MM-DD' strings spanning `days` calendar days.

    end=None: anchor on today (Asia/Shanghai).
    end='YYYY-MM-DD': anchor on the user-supplied date.
    """
    if end is None:
        anchor = dt.datetime.now(tz=MARKET_TZ)
        end = anchor.strftime("%Y-%m-%d")
    else:
        anchor = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=MARKET_TZ)
    start = (anchor - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    return start, end


def scalar(v: object) -> object:
    """Normalize a pandas scalar for JSON: numpy scalar -> Python native, NaN -> None, date -> ISO string."""
    if isinstance(v, np.generic):  # np.int64/bool_ aren't json-native; .item() -> Python native
        v = cast("object", v.item())  # numpy stub types .item() as Any; pin it back to object
    if v is None:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 10)
    if isinstance(v, dt.date):  # date/datetime/pd.Timestamp; pd.NaT is also a dt.date subclass
        return None if pd.isna(v) else v.isoformat()
    return v


def safe_float(val: object) -> float | None:
    """Coerce to float; None/NaN/str-unconvertible -> None.

    Normalizes through scalar() first: raw akshare DataFrame scalars reach here
    unnormalized, and np.int64 is NOT an int subclass, so a bare
    isinstance(val, int) check silently dropped integer columns to None.
    Routing through scalar() keeps the numpy / NaN rules in one place.
    """
    val = scalar(val)
    if isinstance(val, (int, float)):
        return float(val)  # scalar() already mapped NaN -> None, so this is finite
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def as_float(v: object) -> float:
    """Cast to float. Raises on None / NaN / unconvertible."""
    if v is None:
        msg = "value is None"
        raise TypeError(msg)
    if isinstance(v, float):
        if math.isnan(v):
            msg = "value is NaN"
            raise ValueError(msg)
        return v
    if isinstance(v, (int, str)):
        return float(v)
    msg = f"cannot convert to float: {type(v).__name__}"
    raise TypeError(msg)


def df_to_records(df: pd.DataFrame) -> list[Record]:
    """Convert a DataFrame to a list of dicts. NaN -> None, types preserved."""
    return [{str(k): scalar(v) for k, v in record.items()} for record in df.to_dict(orient="records")]
